import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.logger import get_logger
from src.chat.knowledge.embedding_store import EmbeddingManager, EmbeddingItem
from src.chat.knowledge.kg_manager import KGManager

logger = get_logger("问答管理")


@dataclass
class SearchResult:
    """检索结果"""

    content: str
    score: float
    source: str
    metadata: Dict[str, Any]


@dataclass
class KnowledgeContext:
    """知识上下文"""

    question: str
    paragraphs: List[SearchResult]
    entities: List[SearchResult]
    relations: List[Tuple[str, str, str, float]]
    formatted_text: str


def dynamic_select_top_k(
    scores: List[Tuple[Any, float]],
    jump_factor: float = 0.5,
    variance_factor: float = 1.0,
) -> List[Tuple[Any, float, float]]:
    """动态 TopK 选择"""
    if not scores:
        return []
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    max_score = sorted_scores[0][1]
    min_score = sorted_scores[-1][1]
    if max_score == min_score:
        return [(s[0], s[1], 1.0) for s in sorted_scores]
    normalized = []
    for item, score in sorted_scores:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((item, score, norm_score))
    jump_idx = 0
    for i in range(1, len(normalized)):
        curr_diff = abs(normalized[i][2] - normalized[i - 1][2])
        max_diff = (
            abs(normalized[jump_idx][2] - normalized[jump_idx - 1][2])
            if jump_idx > 0
            else 0
        )
        if curr_diff > max_diff:
            jump_idx = i
    jump_threshold = (
        normalized[jump_idx][2] if jump_idx < len(normalized) else 0
    )
    mean_score = sum(s[2] for s in normalized) / len(normalized)
    var_score = sum((s[2] - mean_score) ** 2 for s in normalized) / len(
        normalized
    )
    threshold = jump_factor * jump_threshold + (1 - jump_factor) * (
        mean_score + variance_factor * var_score
    )
    return [s for s in normalized if s[2] > threshold]


class QAManager:
    """问答管理器"""

    def __init__(
        self,
        embed_manager: EmbeddingManager,
        kg_manager: KGManager,
        paragraph_threshold: float = 0.3,
        entity_threshold: float = 0.4,
        relation_threshold: float = 0.5,
        max_paragraphs: int = 5,
        max_entities: int = 10,
        max_relations: int = 10,
    ):
        self.embed_manager = embed_manager
        self.kg_manager = kg_manager
        self.paragraph_threshold = paragraph_threshold
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self.max_paragraphs = max_paragraphs
        self.max_entities = max_entities
        self.max_relations = max_relations

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """获取查询向量"""
        return self.embed_manager.generate_embedding(query)

    def _search_paragraphs(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """搜索段落"""
        results = self.embed_manager.search_paragraphs(query_vector, top_k)
        search_results = []
        for key, score, item in results:
            if score >= self.paragraph_threshold:
                search_results.append(
                    SearchResult(
                        content=item.str,
                        score=score,
                        source="paragraph",
                        metadata=item.metadata,
                    )
                )
        return search_results

    def _search_entities(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """搜索实体"""
        results = self.embed_manager.search_entities(query_vector, top_k)
        search_results = []
        for key, score, item in results:
            if score >= self.entity_threshold:
                search_results.append(
                    SearchResult(
                        content=item.str,
                        score=score,
                        source="entity",
                        metadata=item.metadata,
                    )
                )
        return search_results

    def _search_relations(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """搜索关系"""
        results = self.embed_manager.search_relations(query_vector, top_k)
        search_results = []
        for key, score, item in results:
            if score >= self.relation_threshold:
                search_results.append(
                    SearchResult(
                        content=item.str,
                        score=score,
                        source="relation",
                        metadata=item.metadata,
                    )
                )
        return search_results

    def _kg_retrieve(
        self,
        entities: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, str, str, float]]:
        """知识图谱检索"""
        return self.kg_manager.kg_search(
            query_entities=entities,
            embed_manager=self.embed_manager,
            top_k=top_k,
        )

    def process_query(
        self,
        question: str,
        top_k: int = 5,
    ) -> KnowledgeContext:
        """处理查询"""
        query_vector = self._get_query_embedding(question)
        if query_vector is None:
            logger.warning("[QAManager] 无法生成查询向量")
            return KnowledgeContext(
                question=question,
                paragraphs=[],
                entities=[],
                relations=[],
                formatted_text="",
            )
        paragraphs = self._search_paragraphs(query_vector, self.max_paragraphs)
        entities = self._search_entities(query_vector, self.max_entities)
        entity_names = [e.content for e in entities if e.content]
        kg_relations = self._kg_retrieve(entity_names, self.max_relations)
        relations = []
        for subject, predicate, obj, score in kg_relations:
            relations.append((subject, predicate, obj, score))
        formatted_text = self._format_knowledge(
            paragraphs, entities, relations
        )
        return KnowledgeContext(
            question=question,
            paragraphs=paragraphs,
            entities=entities,
            relations=relations,
            formatted_text=formatted_text,
        )

    def _format_knowledge(
        self,
        paragraphs: List[SearchResult],
        entities: List[SearchResult],
        relations: List[Tuple[str, str, str, float]],
    ) -> str:
        """格式化知识内容"""
        parts = []
        if paragraphs:
            parts.append("【相关段落】")
            for i, p in enumerate(paragraphs, 1):
                parts.append(f"{i}. [相关度:{p.score:.3f}] {p.content[:500]}")
        if entities:
            parts.append("\n【相关实体】")
            entity_strs = [
                f"{e.content}(相关度:{e.score:.3f})" for e in entities[:5]
            ]
            parts.append(", ".join(entity_strs))
        if relations:
            parts.append("\n【知识关系】")
            for i, (s, p, o, score) in enumerate(relations[:5], 1):
                parts.append(f"{i}. {s} --[{p}]--> {o} (权重:{score:.3f})")
        return "\n".join(parts)

    def get_knowledge(
        self,
        question: str,
        top_k: int = 5,
        max_length: int = 2000,
    ) -> str:
        """获取知识内容"""
        context = self.process_query(question, top_k)
        if not context.formatted_text:
            return ""
        result = context.formatted_text
        if len(result) > max_length:
            result = result[:max_length] + "\n...[内容已截断]"
        return result

    async def process_query_async(
        self,
        question: str,
        top_k: int = 5,
    ) -> KnowledgeContext:
        """异步处理查询"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process_query(question, top_k),
        )

    async def get_knowledge_async(
        self,
        question: str,
        top_k: int = 5,
        max_length: int = 2000,
    ) -> str:
        """异步获取知识内容"""
        context = await self.process_query_async(question, top_k)
        if not context.formatted_text:
            return ""
        result = context.formatted_text
        if len(result) > max_length:
            result = result[:max_length] + "\n...[内容已截断]"
        return result

    def search_with_filters(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """带过滤条件的搜索"""
        query_vector = self._get_query_embedding(question)
        if query_vector is None:
            return []
        all_results = []
        pg_results = self._search_paragraphs(query_vector, top_k)
        all_results.extend(pg_results)
        ent_results = self._search_entities(query_vector, top_k)
        all_results.extend(ent_results)
        if filters:
            filtered_results = []
            for r in all_results:
                match = True
                for key, value in filters.items():
                    if key in r.metadata:
                        if r.metadata[key] != value:
                            match = False
                            break
                if match:
                    filtered_results.append(r)
            all_results = filtered_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def get_related_knowledge(
        self,
        entity: str,
        depth: int = 1,
    ) -> List[Tuple[str, str, str]]:
        """获取实体相关知识"""
        relations = self.kg_manager.get_entity_relations(entity)
        neighbors = self.kg_manager.get_entity_neighbors(entity, depth)
        all_relations = list(relations)
        for neighbor in neighbors:
            neighbor_relations = self.kg_manager.get_entity_relations(neighbor)
            for rel in neighbor_relations:
                if rel not in all_relations:
                    all_relations.append(rel)
        return all_relations

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        embed_stats = self.embed_manager.get_stats()
        kg_stats = self.kg_manager.get_stats()
        return {
            "embedding": embed_stats,
            "knowledge_graph": kg_stats,
            "thresholds": {
                "paragraph": self.paragraph_threshold,
                "entity": self.entity_threshold,
                "relation": self.relation_threshold,
            },
        }


class HybridQAManager:
    """混合问答管理器，结合向量检索和知识图谱"""

    def __init__(
        self,
        embed_manager: EmbeddingManager,
        kg_manager: KGManager,
        vector_weight: float = 0.6,
        kg_weight: float = 0.4,
    ):
        self.embed_manager = embed_manager
        self.kg_manager = kg_manager
        self.vector_weight = vector_weight
        self.kg_weight = kg_weight
        self._base_qa = QAManager(embed_manager, kg_manager)

    def hybrid_search(
        self,
        question: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float, str]]:
        """混合检索"""
        query_vector = self.embed_manager.generate_embedding(question)
        if query_vector is None:
            return []
        vector_results = self.embed_manager.search_paragraphs(
            query_vector, top_k * 2
        )
        vector_scores = {}
        for key, score, item in vector_results:
            content = item.str
            if content not in vector_scores or vector_scores[content] < score:
                vector_scores[content] = score
        entity_results = self.embed_manager.search_entities(
            query_vector, top_k
        )
        entity_names = [item.str for _, _, item in entity_results if item]
        kg_triples = self.kg_manager.kg_search(
            entity_names, self.embed_manager, top_k * 2
        )
        kg_scores = {}
        for subject, predicate, obj, score in kg_triples:
            content = f"{subject} {predicate} {obj}"
            if content not in kg_scores or kg_scores[content] < score:
                kg_scores[content] = score
        all_contents = set(vector_scores.keys()) | set(kg_scores.keys())
        combined_results = []
        for content in all_contents:
            v_score = vector_scores.get(content, 0.0)
            k_score = kg_scores.get(content, 0.0)
            combined_score = (
                self.vector_weight * v_score + self.kg_weight * k_score
            )
            combined_results.append((content, combined_score, "hybrid"))
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]

    def get_knowledge(self, question: str, top_k: int = 5) -> str:
        """获取知识"""
        results = self.hybrid_search(question, top_k)
        if not results:
            return ""
        parts = ["【相关知识】"]
        for i, (content, score, source) in enumerate(results, 1):
            parts.append(f"{i}. [相关度:{score:.3f}] {content[:300]}")
        return "\n".join(parts)

    async def get_knowledge_async(self, question: str, top_k: int = 5) -> str:
        """异步获取知识"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_knowledge(question, top_k),
        )


class QAManagerFactory:
    """问答管理器工厂"""

    _instance: Optional[QAManager] = None
    _hybrid_instance: Optional[HybridQAManager] = None

    @classmethod
    def create(
        cls,
        embed_manager: EmbeddingManager,
        kg_manager: KGManager,
        **kwargs,
    ) -> QAManager:
        """创建问答管理器"""
        if cls._instance is None:
            cls._instance = QAManager(embed_manager, kg_manager, **kwargs)
        return cls._instance

    @classmethod
    def create_hybrid(
        cls,
        embed_manager: EmbeddingManager,
        kg_manager: KGManager,
        **kwargs,
    ) -> HybridQAManager:
        """创建混合问答管理器"""
        if cls._hybrid_instance is None:
            cls._hybrid_instance = HybridQAManager(
                embed_manager, kg_manager, **kwargs
            )
        return cls._hybrid_instance

    @classmethod
    def get_instance(cls) -> Optional[QAManager]:
        """获取实例"""
        return cls._instance

    @classmethod
    def reset(cls):
        """重置实例"""
        cls._instance = None
        cls._hybrid_instance = None

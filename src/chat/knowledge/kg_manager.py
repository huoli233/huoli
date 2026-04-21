import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.common.logger import get_logger
from src.chat.knowledge.constants import INVALID_ENTITY as DEFAULT_INVALID_ENTITY

logger = get_logger("知识图谱")


@dataclass
class KGNode:
    """知识图谱节点"""

    name: str
    node_type: str = "entity"
    attributes: Dict[str, Any] = field(default_factory=dict)
    in_edges: List[str] = field(default_factory=list)
    out_edges: List[str] = field(default_factory=list)


@dataclass
class KGEdge:
    """知识图谱边"""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


class DiGraph:
    """有向图数据结构"""

    def __init__(self):
        self._nodes: Dict[str, KGNode] = {}
        self._edges: Dict[str, KGEdge] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, name: str, node_type: str = "entity", **attrs) -> bool:
        """添加节点"""
        if name in self._nodes:
            node = self._nodes[name]
            node.attributes.update(attrs)
            return False
        node = KGNode(name=name, node_type=node_type, attributes=attrs)
        self._nodes[name] = node
        return True

    def remove_node(self, name: str) -> bool:
        """删除节点及其关联边"""
        if name not in self._nodes:
            return False
        edges_to_remove = []
        for edge_id, edge in self._edges.items():
            if edge.source == name or edge.target == name:
                edges_to_remove.append(edge_id)
        for edge_id in edges_to_remove:
            self.remove_edge_by_id(edge_id)
        del self._nodes[name]
        if name in self._adjacency:
            del self._adjacency[name]
        if name in self._reverse_adjacency:
            del self._reverse_adjacency[name]
        return True

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        **attrs,
    ) -> str:
        """添加边，返回边ID"""
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)
        edge_id = f"{source}|{relation}|{target}"
        if edge_id in self._edges:
            edge = self._edges[edge_id]
            edge.weight = weight
            edge.attributes.update(attrs)
            return edge_id
        edge = KGEdge(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            attributes=attrs,
        )
        self._edges[edge_id] = edge
        self._nodes[source].out_edges.append(edge_id)
        self._nodes[target].in_edges.append(edge_id)
        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)
        return edge_id

    def remove_edge(
        self, source: str, target: str, relation: str = None
    ) -> bool:
        """删除边"""
        if relation:
            edge_id = f"{source}|{relation}|{target}"
            return self.remove_edge_by_id(edge_id)
        edges_to_remove = []
        for edge_id, edge in self._edges.items():
            if edge.source == source and edge.target == target:
                edges_to_remove.append(edge_id)
        for edge_id in edges_to_remove:
            self.remove_edge_by_id(edge_id)
        return len(edges_to_remove) > 0

    def remove_edge_by_id(self, edge_id: str) -> bool:
        """通过ID删除边"""
        if edge_id not in self._edges:
            return False
        edge = self._edges[edge_id]
        source, target = edge.source, edge.target
        if edge_id in self._nodes[source].out_edges:
            self._nodes[source].out_edges.remove(edge_id)
        if edge_id in self._nodes[target].in_edges:
            self._nodes[target].in_edges.remove(edge_id)
        self._adjacency[source].discard(target)
        self._reverse_adjacency[target].discard(source)
        del self._edges[edge_id]
        return True

    def get_node(self, name: str) -> Optional[KGNode]:
        """获取节点"""
        return self._nodes.get(name)

    def get_edge(
        self, source: str, target: str, relation: str = None
    ) -> Optional[KGEdge]:
        """获取边"""
        if relation:
            edge_id = f"{source}|{relation}|{target}"
            return self._edges.get(edge_id)
        for _edge_id, edge in self._edges.items():
            if edge.source == source and edge.target == target:
                return edge
        return None

    def get_node_list(self) -> List[str]:
        """获取所有节点名称"""
        return list(self._nodes.keys())

    def get_edge_list(self) -> List[Tuple[str, str, str, float]]:
        """获取所有边"""
        return [
            (e.source, e.target, e.relation, e.weight)
            for e in self._edges.values()
        ]

    def get_neighbors(self, node: str) -> Set[str]:
        """获取邻居节点"""
        return self._adjacency.get(node, set()).copy()

    def get_predecessors(self, node: str) -> Set[str]:
        """获取前驱节点"""
        return self._reverse_adjacency.get(node, set()).copy()

    def get_successors(self, node: str) -> Set[str]:
        """获取后继节点"""
        return self._adjacency.get(node, set()).copy()

    def get_out_edges(self, node: str) -> List[KGEdge]:
        """获取出边"""
        if node not in self._nodes:
            return []
        return [
            self._edges[eid]
            for eid in self._nodes[node].out_edges
            if eid in self._edges
        ]

    def get_in_edges(self, node: str) -> List[KGEdge]:
        """获取入边"""
        if node not in self._nodes:
            return []
        return [
            self._edges[eid]
            for eid in self._nodes[node].in_edges
            if eid in self._edges
        ]

    def node_count(self) -> int:
        """节点数量"""
        return len(self._nodes)

    def edge_count(self) -> int:
        """边数量"""
        return len(self._edges)

    def has_node(self, name: str) -> bool:
        """检查节点是否存在"""
        return name in self._nodes

    def has_edge(self, source: str, target: str) -> bool:
        """检查边是否存在"""
        return target in self._adjacency.get(source, set())


class PageRankCalculator:
    """PageRank 计算器"""

    def __init__(
        self,
        damping: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def calculate(
        self,
        graph: DiGraph,
        personalization: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """计算 PageRank 值"""
        nodes = graph.get_node_list()
        n = len(nodes)
        if n == 0:
            return {}
        node_idx = {node: i for i, node in enumerate(nodes)}
        out_degree = np.zeros(n)
        adj_matrix = np.zeros((n, n))
        for node in nodes:
            idx = node_idx[node]
            neighbors = graph.get_successors(node)
            out_degree[idx] = len(neighbors) if neighbors else 0
            for neighbor in neighbors:
                if neighbor in node_idx:
                    adj_matrix[node_idx[neighbor], idx] = 1.0
        for i in range(n):
            if out_degree[i] > 0:
                adj_matrix[:, i] /= out_degree[i]
            else:
                adj_matrix[:, i] = 1.0 / n
        pr = np.ones(n) / n
        if personalization:
            p_vector = np.zeros(n)
            total_p = sum(personalization.values())
            for node, score in personalization.items():
                if node in node_idx:
                    p_vector[node_idx[node]] = (
                        score / total_p if total_p > 0 else 1.0 / n
                    )
            if p_vector.sum() == 0:
                p_vector = np.ones(n) / n
        else:
            p_vector = np.ones(n) / n
        for iteration in range(self.max_iterations):
            new_pr = (
                1 - self.damping
            ) * p_vector + self.damping * adj_matrix @ pr
            diff = np.abs(new_pr - pr).sum()
            pr = new_pr
            if diff < self.tolerance:
                logger.debug(f"PageRank 收敛于第 {iteration + 1} 次迭代")
                break
        return {node: float(pr[node_idx[node]]) for node in nodes}

    def personalized_pagerank(
        self,
        graph: DiGraph,
        seed_nodes: List[str],
        seed_weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """个性化 PageRank"""
        if not seed_nodes:
            return self.calculate(graph)
        personalization = {}
        if seed_weights and len(seed_weights) == len(seed_nodes):
            for node, weight in zip(seed_nodes, seed_weights, strict=True):
                personalization[node] = weight
        else:
            weight = 1.0 / len(seed_nodes)
            for node in seed_nodes:
                personalization[node] = weight
        return self.calculate(graph, personalization)


class KGManager:
    """知识图谱管理器"""

    INVALID_ENTITY = list(DEFAULT_INVALID_ENTITY)

    def __init__(self, data_dir: str = ""):
        self.data_dir = data_dir or self._get_default_data_dir()
        self.graph = DiGraph()
        self.stored_paragraph_hashes: Set[str] = set()
        self.ent_appear_cnt: Dict[str, int] = defaultdict(int)
        self._pagerank = PageRankCalculator()
        self._lock = threading.RLock()
        os.makedirs(self.data_dir, exist_ok=True)
        self.graph_file = os.path.join(self.data_dir, "knowledge_graph.json")
        self.hash_file = os.path.join(self.data_dir, "paragraph_hashes.json")
        self.cnt_file = os.path.join(self.data_dir, "entity_counts.json")

    def _get_default_data_dir(self) -> str:
        """获取默认数据目录"""
        current = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(current, "..", "..", ".."))
        return os.path.join(root, "data", "knowledge")

    def _is_valid_entity(self, entity: str) -> bool:
        """检查实体是否有效"""
        if not entity or not isinstance(entity, str):
            return False
        entity = entity.strip()
        if not entity or entity in self.INVALID_ENTITY:
            return False
        if len(entity) < 2 or len(entity) > 50:
            return False
        return True

    def _normalize_entity(self, entity: str) -> str:
        """规范化实体名称"""
        return entity.strip().replace("\n", " ").replace("\t", " ")

    def build_kg(
        self,
        triple_list_data: Dict[str, List[List[str]]],
        embed_manager: Optional[Any] = None,
    ) -> Dict[str, int]:
        """构建知识图谱"""
        stats = {"nodes": 0, "edges": 0, "paragraphs": 0}
        with self._lock:
            for pg_hash, triples in triple_list_data.items():
                if pg_hash in self.stored_paragraph_hashes:
                    continue
                self.stored_paragraph_hashes.add(pg_hash)
                stats["paragraphs"] += 1
                for triple in triples:
                    if len(triple) != 3:
                        continue
                    subject, predicate, obj = triple
                    subject = self._normalize_entity(subject)
                    predicate = self._normalize_entity(predicate)
                    obj = self._normalize_entity(obj)
                    if not self._is_valid_entity(subject):
                        continue
                    if not self._is_valid_entity(obj):
                        continue
                    if not predicate:
                        continue
                    if self.graph.add_node(subject, "entity"):
                        stats["nodes"] += 1
                    if self.graph.add_node(obj, "entity"):
                        stats["nodes"] += 1
                    edge_id = self.graph.add_edge(subject, obj, predicate)
                    if edge_id:
                        stats["edges"] += 1
                    self.ent_appear_cnt[subject] += 1
                    self.ent_appear_cnt[obj] += 1
            self._synonym_connect(embed_manager)
        logger.info(f"[KGManager] 知识图谱构建完成: {stats}")
        return stats

    def _synonym_connect(self, embed_manager: Optional[Any] = None):
        """连接同义实体"""
        if embed_manager is None:
            return
        nodes = self.graph.get_node_list()
        entities = [
            n for n in nodes if self.graph.get_node(n).node_type == "entity"
        ]
        if len(entities) < 2:
            return
        logger.info(
            f"[KGManager] 正在处理 {len(entities)} 个实体的同义连接..."
        )
        try:
            entity_vectors = {}
            for entity in entities:
                key = f"entity-{entity}"
                item = embed_manager.entities_embedding_store.get_item(key)
                if item and item.vector is not None:
                    entity_vectors[entity] = item.vector
            if len(entity_vectors) < 2:
                return
            connected = 0
            entity_list = list(entity_vectors.keys())
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    e1, e2 = entity_list[i], entity_list[j]
                    v1, v2 = entity_vectors[e1], entity_vectors[e2]
                    similarity = float(np.dot(v1, v2))
                    if similarity > 0.95:
                        if not self.graph.has_edge(e1, e2):
                            self.graph.add_edge(e1, e2, "synonym", similarity)
                            connected += 1
                        if not self.graph.has_edge(e2, e1):
                            self.graph.add_edge(e2, e1, "synonym", similarity)
                            connected += 1
            logger.info(f"[KGManager] 同义连接完成，新增 {connected} 条边")
        except Exception as e:
            logger.error(f"[KGManager] 同义连接失败: {e}")

    def kg_search(
        self,
        query_entities: List[str],
        embed_manager: Optional[Any] = None,
        top_k: int = 10,
        ppr_top_k: int = 20,
    ) -> List[Tuple[str, str, str, float]]:
        """知识图谱检索"""
        if not query_entities:
            return []
        valid_seeds = []
        seed_weights = []
        for entity in query_entities:
            entity = self._normalize_entity(entity)
            if self.graph.has_node(entity):
                valid_seeds.append(entity)
                seed_weights.append(self.ent_appear_cnt.get(entity, 1))
        if not valid_seeds:
            return []
        pr_scores = self._pagerank.personalized_pagerank(
            self.graph, valid_seeds, seed_weights
        )
        sorted_nodes = sorted(
            pr_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_nodes = sorted_nodes[:ppr_top_k]
        results = []
        seen_triples = set()
        for node, score in top_nodes:
            out_edges = self.graph.get_out_edges(node)
            for edge in out_edges:
                triple_key = (edge.source, edge.relation, edge.target)
                if triple_key not in seen_triples:
                    results.append(
                        (
                            edge.source,
                            edge.relation,
                            edge.target,
                            score * edge.weight,
                        )
                    )
                    seen_triples.add(triple_key)
            if len(results) >= top_k:
                break
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def delete_paragraphs(
        self,
        pg_hashes: List[str],
        ent_hashes: Optional[List[str]] = None,
        remove_orphan_entities: bool = True,
    ) -> Dict[str, int]:
        """删除段落及相关数据"""
        stats = {"paragraphs": 0, "edges": 0, "entities": 0}
        with self._lock:
            for pg_hash in pg_hashes:
                if pg_hash in self.stored_paragraph_hashes:
                    self.stored_paragraph_hashes.remove(pg_hash)
                    stats["paragraphs"] += 1
            if remove_orphan_entities:
                all_nodes = set(self.graph.get_node_list())
                connected_nodes = set()
                for edge in self.graph._edges.values():
                    connected_nodes.add(edge.source)
                    connected_nodes.add(edge.target)
                orphan_nodes = all_nodes - connected_nodes
                for node in orphan_nodes:
                    if self.graph.get_node(node).node_type == "entity":
                        self.graph.remove_node(node)
                        stats["entities"] += 1
        logger.info(f"[KGManager] 删除完成: {stats}")
        return stats

    def get_entity_relations(self, entity: str) -> List[Tuple[str, str, str]]:
        """获取实体的所有关系"""
        if not self.graph.has_node(entity):
            return []
        relations = []
        out_edges = self.graph.get_out_edges(entity)
        for edge in out_edges:
            relations.append((edge.source, edge.relation, edge.target))
        in_edges = self.graph.get_in_edges(entity)
        for edge in in_edges:
            relations.append((edge.source, edge.relation, edge.target))
        return relations

    def get_entity_neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """获取实体的邻居节点"""
        if not self.graph.has_node(entity):
            return []
        visited = {entity}
        current_level = {entity}
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                neighbors = self.graph.get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
            if not current_level:
                break
        return list(visited - {entity})

    def get_entity_count(self, entity: str) -> int:
        """获取实体出现次数"""
        return self.ent_appear_cnt.get(entity, 0)

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "nodes": self.graph.node_count(),
            "edges": self.graph.edge_count(),
            "paragraphs": len(self.stored_paragraph_hashes),
            "entities": len(self.ent_appear_cnt),
        }

    def save_to_file(self) -> bool:
        """保存到文件"""
        with self._lock:
            try:
                graph_data = {
                    "nodes": [],
                    "edges": [],
                }
                for name, node in self.graph._nodes.items():
                    graph_data["nodes"].append(
                        {
                            "name": name,
                            "type": node.node_type,
                            "attributes": node.attributes,
                        }
                    )
                for _edge_id, edge in self.graph._edges.items():
                    graph_data["edges"].append(
                        {
                            "source": edge.source,
                            "target": edge.target,
                            "relation": edge.relation,
                            "weight": edge.weight,
                            "attributes": edge.attributes,
                        }
                    )
                with open(self.graph_file, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)
                with open(self.hash_file, "w", encoding="utf-8") as f:
                    json.dump(list(self.stored_paragraph_hashes), f)
                with open(self.cnt_file, "w", encoding="utf-8") as f:
                    json.dump(dict(self.ent_appear_cnt), f)
                logger.info(f"[KGManager] 保存完成: {self.get_stats()}")
                return True
            except Exception as e:
                logger.error(f"[KGManager] 保存失败: {e}")
                return False

    def load_from_file(self) -> bool:
        """从文件加载"""
        with self._lock:
            try:
                if os.path.exists(self.graph_file):
                    with open(self.graph_file, "r", encoding="utf-8") as f:
                        graph_data = json.load(f)
                    for node_data in graph_data.get("nodes", []):
                        self.graph.add_node(
                            node_data["name"],
                            node_data.get("type", "entity"),
                            **node_data.get("attributes", {}),
                        )
                    for edge_data in graph_data.get("edges", []):
                        self.graph.add_edge(
                            edge_data["source"],
                            edge_data["target"],
                            edge_data["relation"],
                            edge_data.get("weight", 1.0),
                            **edge_data.get("attributes", {}),
                        )
                if os.path.exists(self.hash_file):
                    with open(self.hash_file, "r", encoding="utf-8") as f:
                        self.stored_paragraph_hashes = set(json.load(f))
                if os.path.exists(self.cnt_file):
                    with open(self.cnt_file, "r", encoding="utf-8") as f:
                        self.ent_appear_cnt = defaultdict(int, json.load(f))
                logger.info(f"[KGManager] 加载完成: {self.get_stats()}")
                return True
            except Exception as e:
                logger.error(f"[KGManager] 加载失败: {e}")
                return False

    def clear(self):
        """清空知识图谱"""
        with self._lock:
            self.graph = DiGraph()
            self.stored_paragraph_hashes.clear()
            self.ent_appear_cnt.clear()
            logger.info("[KGManager] 知识图谱已清空")

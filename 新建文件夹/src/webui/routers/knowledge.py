import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Cookie, Depends, Header, Query
from pydantic import BaseModel

from src.config.config import global_config
from src.webui.core import verify_auth_token_from_cookie_or_header

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webui/knowledge", tags=["knowledge"])

_paragraph_store_cache = None


def _get_paragraph_store():
    """延迟加载段落 embedding store"""
    if not global_config.webui.enable_paragraph_content:
        return None
    global _paragraph_store_cache
    if _paragraph_store_cache is not None:
        return _paragraph_store_cache
    try:
        from src.chat.knowledge.embedding_store import EmbeddingStore

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
        embedding_dir = os.path.join(root_path, "data/embedding")
        paragraph_store = EmbeddingStore(
            store_name="paragraph",
            data_dir=embedding_dir,
        )
        paragraph_store.load_from_file()
        _paragraph_store_cache = paragraph_store
        logger.info(
            f"成功加载段落 embedding store，包含 {paragraph_store.count()} 个段落"
        )
        return paragraph_store
    except Exception as e:
        logger.warning(f"加载段落 embedding store 失败: {e}")
        return None


def _get_paragraph_content(node_id: str) -> tuple:
    """从 embedding store 获取段落完整内容"""
    try:
        paragraph_store = _get_paragraph_store()
        if paragraph_store is None:
            return None, False
        paragraph_item = paragraph_store.get_item(node_id)
        if paragraph_item is not None:
            content = paragraph_item.str
            if content:
                return content, True
        return None, True
    except Exception as e:
        logger.debug(f"获取段落内容失败: {e}")
        return None, True


def require_auth(
    huoli_session: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None),
) -> bool:
    """认证依赖"""
    return verify_auth_token_from_cookie_or_header(
        huoli_session, authorization
    )


class KnowledgeNode(BaseModel):
    """知识节点"""

    id: str
    type: str
    content: str
    create_time: Optional[float] = None


class KnowledgeEdge(BaseModel):
    """知识边"""

    source: str
    target: str
    weight: float
    create_time: Optional[float] = None
    update_time: Optional[float] = None


class KnowledgeGraph(BaseModel):
    """知识图谱"""

    nodes: List[KnowledgeNode]
    edges: List[KnowledgeEdge]


class KnowledgeStats(BaseModel):
    """知识库统计信息"""

    total_nodes: int
    total_edges: int
    entity_nodes: int
    paragraph_nodes: int
    avg_connections: float


def _load_kg_manager():
    """延迟加载 KGManager"""
    try:
        from src.chat.knowledge.kg_manager import KGManager

        kg_manager = KGManager()
        kg_manager.load_from_file()
        return kg_manager
    except Exception as e:
        logger.error(f"加载 KGManager 失败: {e}")
        return None


def _convert_graph_to_json(kg_manager) -> KnowledgeGraph:
    """将 DiGraph 转换为 JSON 格式"""
    if kg_manager is None or kg_manager.graph is None:
        return KnowledgeGraph(nodes=[], edges=[])
    graph = kg_manager.graph
    nodes = []
    edges = []
    node_list = graph.get_node_list()
    for node_id in node_list:
        try:
            node_data = graph.get_node(node_id)
            if node_data is None:
                continue
            node_type = (
                "entity" if node_data.node_type == "entity" else "paragraph"
            )
            if node_type == "paragraph":
                full_content, _ = _get_paragraph_content(node_id)
                content = full_content if full_content is not None else node_id
            else:
                content = node_id
            create_time = node_data.attributes.get("create_time")
            nodes.append(
                KnowledgeNode(
                    id=node_id,
                    type=node_type,
                    content=content,
                    create_time=create_time,
                )
            )
        except Exception as e:
            logger.warning(f"跳过节点 {node_id}: {e}")
            continue
    edge_list = graph.get_edge_list()
    for edge_tuple in edge_list:
        try:
            source, target, relation, weight = edge_tuple
            edges.append(
                KnowledgeEdge(
                    source=source,
                    target=target,
                    weight=weight,
                    create_time=None,
                    update_time=None,
                )
            )
        except Exception as e:
            logger.warning(f"跳过边 {edge_tuple}: {e}")
            continue
    return KnowledgeGraph(nodes=nodes, edges=edges)


@router.get("/graph", response_model=KnowledgeGraph)
async def get_knowledge_graph(
    limit: int = Query(100, ge=1, le=10000, description="返回的最大节点数"),
    node_type: str = Query("all", description="节点类型过滤"),
    _auth: bool = Depends(require_auth),
):
    """获取知识图谱"""
    try:
        kg_manager = _load_kg_manager()
        if kg_manager is None:
            logger.warning("KGManager 未初始化，返回空图谱")
            return KnowledgeGraph(nodes=[], edges=[])
        graph = kg_manager.graph
        all_node_list = graph.get_node_list()
        if node_type == "entity":
            all_node_list = [
                n
                for n in all_node_list
                if graph.get_node(n)
                and graph.get_node(n).node_type == "entity"
            ]
        elif node_type == "paragraph":
            all_node_list = [
                n
                for n in all_node_list
                if graph.get_node(n)
                and graph.get_node(n).node_type != "entity"
            ]
        total_nodes = len(all_node_list)
        if len(all_node_list) > limit:
            node_list = all_node_list[:limit]
        else:
            node_list = all_node_list
        logger.info(f"总节点数: {total_nodes}, 返回节点: {len(node_list)}")
        nodes = []
        node_ids = set()
        for node_id in node_list:
            try:
                node_data = graph.get_node(node_id)
                if node_data is None:
                    continue
                node_type_val = (
                    "entity"
                    if node_data.node_type == "entity"
                    else "paragraph"
                )
                if node_type_val == "paragraph":
                    full_content, _ = _get_paragraph_content(node_id)
                    content = (
                        full_content if full_content is not None else node_id
                    )
                else:
                    content = node_id
                create_time = node_data.attributes.get("create_time")
                nodes.append(
                    KnowledgeNode(
                        id=node_id,
                        type=node_type_val,
                        content=content,
                        create_time=create_time,
                    )
                )
                node_ids.add(node_id)
            except Exception as e:
                logger.warning(f"跳过节点 {node_id}: {e}")
                continue
        edges = []
        edge_list = graph.get_edge_list()
        for edge_tuple in edge_list:
            try:
                source, target, relation, weight = edge_tuple
                if source not in node_ids or target not in node_ids:
                    continue
                edges.append(
                    KnowledgeEdge(
                        source=source,
                        target=target,
                        weight=weight,
                        create_time=None,
                        update_time=None,
                    )
                )
            except Exception as e:
                logger.warning(f"跳过边 {edge_tuple}: {e}")
                continue
        graph_data = KnowledgeGraph(nodes=nodes, edges=edges)
        logger.info(f"返回知识图谱: {len(nodes)} 个节点, {len(edges)} 条边")
        return graph_data
    except Exception as e:
        logger.error(f"获取知识图谱失败: {e}", exc_info=True)
        return KnowledgeGraph(nodes=[], edges=[])


@router.get("/stats", response_model=KnowledgeStats)
async def get_knowledge_stats(_auth: bool = Depends(require_auth)):
    """获取知识库统计信息"""
    try:
        kg_manager = _load_kg_manager()
        if kg_manager is None or kg_manager.graph is None:
            return KnowledgeStats(
                total_nodes=0,
                total_edges=0,
                entity_nodes=0,
                paragraph_nodes=0,
                avg_connections=0.0,
            )
        graph = kg_manager.graph
        node_list = graph.get_node_list()
        edge_list = graph.get_edge_list()
        total_nodes = len(node_list)
        total_edges = len(edge_list)
        entity_nodes = 0
        paragraph_nodes = 0
        for node_id in node_list:
            try:
                node_data = graph.get_node(node_id)
                if node_data is None:
                    continue
                if node_data.node_type == "entity":
                    entity_nodes += 1
                else:
                    paragraph_nodes += 1
            except Exception:
                continue
        avg_connections = (
            (total_edges * 2) / total_nodes if total_nodes > 0 else 0.0
        )
        return KnowledgeStats(
            total_nodes=total_nodes,
            total_edges=total_edges,
            entity_nodes=entity_nodes,
            paragraph_nodes=paragraph_nodes,
            avg_connections=round(avg_connections, 2),
        )
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}", exc_info=True)
        return KnowledgeStats(
            total_nodes=0,
            total_edges=0,
            entity_nodes=0,
            paragraph_nodes=0,
            avg_connections=0.0,
        )


@router.get("/search", response_model=List[KnowledgeNode])
async def search_knowledge_node(
    query: str = Query(..., min_length=1),
    _auth: bool = Depends(require_auth),
):
    """搜索知识节点"""
    try:
        kg_manager = _load_kg_manager()
        if kg_manager is None or kg_manager.graph is None:
            return []
        graph = kg_manager.graph
        node_list = graph.get_node_list()
        results = []
        query_lower = query.lower()
        for node_id in node_list:
            try:
                node_data = graph.get_node(node_id)
                if node_data is None:
                    continue
                node_type = (
                    "entity"
                    if node_data.node_type == "entity"
                    else "paragraph"
                )
                if node_type == "paragraph":
                    full_content, _ = _get_paragraph_content(node_id)
                    content = (
                        full_content if full_content is not None else node_id
                    )
                else:
                    content = node_id
                if (
                    query_lower in content.lower()
                    or query_lower in node_id.lower()
                ):
                    create_time = node_data.attributes.get("create_time")
                    results.append(
                        KnowledgeNode(
                            id=node_id,
                            type=node_type,
                            content=content,
                            create_time=create_time,
                        )
                    )
            except Exception:
                continue
        logger.info(f"搜索 '{query}' 找到 {len(results)} 个节点")
        return results[:50]
    except Exception as e:
        logger.error(f"搜索节点失败: {e}", exc_info=True)
        return []

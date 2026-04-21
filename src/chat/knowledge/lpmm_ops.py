import asyncio
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from src.common.logger import get_logger
from src.chat.knowledge.embedding_store import EmbeddingManager
from src.chat.knowledge.kg_manager import KGManager
from src.chat.knowledge.qa_manager import QAManager

logger = get_logger("LPMM操作")


class LPMMOperations:
    """LPMM 内部操作接口，封装知识库核心操作"""

    def __init__(self):
        self._initialized = False
        self._embed_manager: Optional[EmbeddingManager] = None
        self._kg_manager: Optional[KGManager] = None
        self._qa_manager: Optional[QAManager] = None

    def initialize(
        self,
        embed_manager: EmbeddingManager,
        kg_manager: KGManager,
        qa_manager: QAManager,
    ):
        """初始化管理器"""
        self._embed_manager = embed_manager
        self._kg_manager = kg_manager
        self._qa_manager = qa_manager
        self._initialized = True
        logger.info("[LPMMOperations] 初始化完成")

    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    async def _run_in_executor(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """在线程池中执行同步操作"""
        loop = asyncio.get_running_loop()
        if kwargs:
            func = partial(func, **kwargs)
        return await loop.run_in_executor(None, func, *args)

    def _ensure_initialized(self):
        """确保已初始化"""
        if not self._initialized:
            raise RuntimeError("LPMM 未初始化，请先调用 initialize()")

    async def add_content(
        self,
        text: str,
        auto_split: bool = True,
        llm_ner: Optional[Any] = None,
        llm_rdf: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """向知识库添加内容"""
        self._ensure_initialized()
        try:
            if auto_split:
                paragraphs = [
                    p.strip() for p in text.split("\n\n") if p.strip()
                ]
            else:
                text_stripped = text.strip()
                if not text_stripped:
                    return {"status": "error", "message": "文本内容为空"}
                paragraphs = [text_stripped]
            if not paragraphs:
                return {"status": "error", "message": "文本内容为空"}
            if llm_ner is None or llm_rdf is None:
                return {"status": "error", "message": "未提供 LLM 客户端"}
            from src.chat.knowledge.ie_process import IEProcess

            ie_process = IEProcess(llm_ner, llm_rdf)
            logger.info(
                f"[LPMM] 正在对 {len(paragraphs)} 段文本执行信息抽取..."
            )
            extracted_docs = await ie_process.process_paragraphs(
                paragraphs
            )
            raw_paragraphs = {}
            triple_list_data = {}
            for doc in extracted_docs:
                if not doc.get("success", True):
                    continue
                pg_hash = str(doc.get("idx", "") or "").strip()
                passage = str(doc.get("passage", "") or "").strip()
                if not pg_hash or not passage:
                    logger.debug(f"[LPMM] 跳过结构不完整的抽取结果: {doc}")
                    continue
                raw_paragraphs[pg_hash] = passage
                triple_list_data[pg_hash] = doc.get("extracted_triples", []) or []
            if not raw_paragraphs:
                return {
                    "status": "error",
                    "count": 0,
                    "message": "信息抽取未产出有效知识",
                }
            new_raw_paragraphs = {}
            new_triple_list_data = {}
            for pg_hash, passage in raw_paragraphs.items():
                key = f"paragraph-{pg_hash}"
                if key not in self._embed_manager.stored_pg_hashes:
                    new_raw_paragraphs[pg_hash] = passage
                    new_triple_list_data[pg_hash] = triple_list_data[pg_hash]
            if not new_raw_paragraphs:
                return {
                    "status": "success",
                    "count": 0,
                    "message": "内容已存在，无需重复导入",
                }
            await self._run_in_executor(
                self._embed_manager.store_new_data_set,
                new_raw_paragraphs,
                new_triple_list_data,
            )
            await self._run_in_executor(
                self._kg_manager.build_kg,
                new_triple_list_data,
                self._embed_manager,
            )
            await self._run_in_executor(
                self._embed_manager.rebuild_faiss_index
            )
            await self._run_in_executor(self._embed_manager.save_to_file)
            await self._run_in_executor(self._kg_manager.save_to_file)
            return {
                "status": "success",
                "count": len(new_raw_paragraphs),
                "message": f"成功导入 {len(new_raw_paragraphs)} 条知识",
            }
        except asyncio.CancelledError:
            logger.warning("[LPMM] 导入操作被用户中断")
            return {"status": "cancelled", "message": "导入操作已被用户中断"}
        except Exception as e:
            logger.error(f"[LPMM] 导入知识失败: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """检索知识库"""
        self._ensure_initialized()
        try:
            knowledge = await self._run_in_executor(
                self._qa_manager.get_knowledge,
                query,
                top_k,
            )
            return [knowledge] if knowledge else []
        except Exception as e:
            logger.error(f"[LPMM] 检索知识失败: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """混合检索知识库"""
        self._ensure_initialized()
        try:
            from src.chat.knowledge.qa_manager import HybridQAManager

            hybrid_qa = HybridQAManager(
                self._embed_manager,
                self._kg_manager,
            )
            results = await self._run_in_executor(
                hybrid_qa.hybrid_search,
                query,
                top_k,
            )
            return [r[0] for r in results]
        except Exception as e:
            logger.error(f"[LPMM] 混合检索失败: {e}")
            return []

    async def delete(
        self,
        keyword: str,
        exact_match: bool = False,
    ) -> Dict[str, Any]:
        """根据关键词删除知识"""
        self._ensure_initialized()
        try:
            to_delete_keys = []
            to_delete_hashes = []
            for (
                key,
                item,
            ) in self._embed_manager.paragraphs_embedding_store.store.items():
                if exact_match:
                    if item.str.strip() == keyword.strip():
                        to_delete_keys.append(key)
                        to_delete_hashes.append(
                            key.replace("paragraph-", "", 1)
                        )
                else:
                    if keyword in item.str:
                        to_delete_keys.append(key)
                        to_delete_hashes.append(
                            key.replace("paragraph-", "", 1)
                        )
            if not to_delete_keys:
                match_type = "完整文段" if exact_match else "关键词"
                return {
                    "status": "info",
                    "deleted_count": 0,
                    "message": f"未找到匹配的内容（{match_type}匹配）",
                }
            deleted_count, _ = await self._run_in_executor(
                self._embed_manager.paragraphs_embedding_store.delete_items,
                to_delete_keys,
            )
            self._embed_manager.stored_pg_hashes = set(
                self._embed_manager.paragraphs_embedding_store.get_all_keys()
            )
            delete_func = partial(
                self._kg_manager.delete_paragraphs,
                to_delete_hashes,
                ent_hashes=None,
                remove_orphan_entities=True,
            )
            await self._run_in_executor(delete_func)
            await self._run_in_executor(
                self._embed_manager.rebuild_faiss_index
            )
            await self._run_in_executor(self._embed_manager.save_to_file)
            await self._run_in_executor(self._kg_manager.save_to_file)
            match_type = "完整文段" if exact_match else "关键词"
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "message": f"已成功删除 {deleted_count} 条相关知识（{match_type}匹配）",
            }
        except asyncio.CancelledError:
            logger.warning("[LPMM] 删除操作被用户中断")
            return {"status": "cancelled", "message": "删除操作已被用户中断"}
        except Exception as e:
            logger.error(f"[LPMM] 删除知识失败: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def clear_all(self) -> Dict[str, Any]:
        """清空整个知识库"""
        self._ensure_initialized()
        try:
            before_stats = {
                "paragraphs": self._embed_manager.paragraphs_embedding_store.count(),
                "entities": self._embed_manager.entities_embedding_store.count(),
                "relations": self._embed_manager.relation_embedding_store.count(),
                "kg_nodes": self._kg_manager.graph.node_count(),
                "kg_edges": self._kg_manager.graph.edge_count(),
            }
            self._embed_manager.clear_all()
            self._kg_manager.clear()
            await self._run_in_executor(self._embed_manager.save_to_file)
            await self._run_in_executor(self._kg_manager.save_to_file)
            after_stats = {
                "paragraphs": 0,
                "entities": 0,
                "relations": 0,
                "kg_nodes": 0,
                "kg_edges": 0,
            }
            return {
                "status": "success",
                "message": "已成功清空LPMM知识库",
                "stats": {
                    "before": before_stats,
                    "after": after_stats,
                },
            }
        except asyncio.CancelledError:
            logger.warning("[LPMM] 清空操作被用户中断")
            return {"status": "cancelled", "message": "清空操作已被用户中断"}
        except Exception as e:
            logger.error(f"[LPMM] 清空知识库失败: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        self._ensure_initialized()
        return {
            "embedding": self._embed_manager.get_stats(),
            "knowledge_graph": self._kg_manager.get_stats(),
        }

    def get_entity_relations(self, entity: str) -> List[tuple]:
        """获取实体的所有关系"""
        self._ensure_initialized()
        return self._kg_manager.get_entity_relations(entity)

    def get_entity_neighbors(self, entity: str, depth: int = 1) -> List[str]:
        """获取实体的邻居节点"""
        self._ensure_initialized()
        return self._kg_manager.get_entity_neighbors(entity, depth)


lpmm_ops = LPMMOperations()


def get_lpmm_ops() -> LPMMOperations:
    """获取 LPMM 操作实例"""
    return lpmm_ops


def initialize_lpmm(
    embed_manager: EmbeddingManager,
    kg_manager: KGManager,
    qa_manager: QAManager,
):
    """初始化 LPMM"""
    lpmm_ops.initialize(embed_manager, kg_manager, qa_manager)

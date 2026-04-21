import asyncio
import json
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.logger import get_logger

logger = get_logger("向量存储")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装，向量检索功能将受限")


@dataclass
class EmbeddingItem:
    """向量存储项"""

    key: str
    vector: Optional[np.ndarray] = None
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    create_time: float = 0.0
    update_time: float = 0.0


class EmbeddingStore:
    """基于 FAISS 的向量存储器"""

    def __init__(
        self,
        store_name: str,
        data_dir: str,
        dimension: int = 1024,
        normalize: bool = True,
    ):
        self.store_name = store_name
        self.data_dir = data_dir
        self.dimension = dimension
        self.normalize = normalize
        self.store: Dict[str, EmbeddingItem] = {}
        self.faiss_index: Optional[Any] = None
        self.idx2hash: Dict[int, str] = {}
        self.hash2idx: Dict[str, int] = {}
        self.dirty = False
        self._lock = threading.RLock()
        os.makedirs(data_dir, exist_ok=True)
        self.store_file = os.path.join(data_dir, f"{store_name}_store.pkl")
        self.index_file_path = os.path.join(
            data_dir, f"{store_name}_index.faiss"
        )
        self.idx2hash_file_path = os.path.join(
            data_dir, f"{store_name}_idx2hash.pkl"
        )

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """向量归一化"""
        if not self.normalize:
            return vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def _build_faiss_index(self) -> bool:
        """构建 FAISS 索引"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 不可用，跳过索引构建")
            return False
        vectors = []
        keys = []
        for key, item in self.store.items():
            if item.vector is not None:
                vectors.append(item.vector)
                keys.append(key)
        if not vectors:
            self.faiss_index = None
            self.idx2hash = {}
            self.hash2idx = {}
            return True
        vectors_array = np.array(vectors, dtype=np.float32)
        n_vectors = len(vectors)
        if n_vectors == 0:
            return True
        try:
            if n_vectors < 100:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
            else:
                nlist = min(int(np.sqrt(n_vectors)), 100)
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.faiss_index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist
                )
                self.faiss_index.train(vectors_array)
            self.faiss_index.add(vectors_array)
            self.idx2hash = {i: keys[i] for i in range(len(keys))}
            self.hash2idx = {keys[i]: i for i in range(len(keys))}
            logger.info(
                f"[{self.store_name}] FAISS 索引构建完成，共 {n_vectors} 个向量"
            )
            return True
        except Exception as e:
            logger.error(f"[{self.store_name}] FAISS 索引构建失败: {e}")
            self.faiss_index = None
            return False

    def rebuild_faiss_index(self) -> bool:
        """重建 FAISS 索引"""
        with self._lock:
            return self._build_faiss_index()

    def add_item(
        self,
        key: str,
        vector: np.ndarray,
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """添加单个向量项"""
        import time

        with self._lock:
            if vector.shape[0] != self.dimension:
                logger.error(
                    f"[{self.store_name}] 向量维度不匹配: {vector.shape[0]} != {self.dimension}"
                )
                return False
            normalized_vector = self._normalize_vector(
                vector.astype(np.float32)
            )
            current_time = time.time()
            if key in self.store:
                item = self.store[key]
                item.vector = normalized_vector
                item.content = text
                item.metadata = metadata or {}
                item.update_time = current_time
            else:
                item = EmbeddingItem(
                    key=key,
                    vector=normalized_vector,
                    str=text,
                    metadata=metadata or {},
                    create_time=current_time,
                    update_time=current_time,
                )
                self.store[key] = item
            self.dirty = True
            return True

    def add_items_batch(
        self,
        items: List[Tuple[str, np.ndarray, str, Dict[str, Any]]],
    ) -> int:
        """批量添加向量项"""
        success_count = 0
        for key, vector, text, metadata in items:
            if self.add_item(key, vector, text, metadata):
                success_count += 1
        return success_count

    def get_item(self, key: str) -> Optional[EmbeddingItem]:
        """获取单个向量项"""
        return self.store.get(key)

    def delete_items(self, keys: List[str]) -> Tuple[int, List[str]]:
        """删除向量项"""
        deleted_count = 0
        deleted_keys = []
        with self._lock:
            for key in keys:
                if key in self.store:
                    del self.store[key]
                    deleted_count += 1
                    deleted_keys.append(key)
                    self.dirty = True
        return deleted_count, deleted_keys

    def search_top_k(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float, EmbeddingItem]]:
        """向量相似度搜索"""
        if len(self.store) == 0:
            return []
        if query_vector.shape[0] != self.dimension:
            logger.error(
                f"[{self.store_name}] 查询向量维度不匹配: {query_vector.shape[0]} != {self.dimension}"
            )
            return []
        normalized_query = self._normalize_vector(
            query_vector.astype(np.float32)
        )
        if self.faiss_index is not None and FAISS_AVAILABLE:
            return self._search_with_faiss(normalized_query, top_k)
        return self._search_brute_force(normalized_query, top_k)

    def _search_with_faiss(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[str, float, EmbeddingItem]]:
        """使用 FAISS 索引搜索"""
        try:
            query_array = query_vector.reshape(1, -1).astype(np.float32)
            actual_k = min(top_k, len(self.store))
            if actual_k == 0:
                return []
            scores, indices = self.faiss_index.search(query_array, actual_k)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx >= 0 and idx in self.idx2hash:
                    key = self.idx2hash[idx]
                    score = float(scores[0][i])
                    item = self.store.get(key)
                    if item:
                        results.append((key, score, item))
            return results
        except Exception as e:
            logger.error(f"[{self.store_name}] FAISS 搜索失败: {e}")
            return self._search_brute_force(query_vector, top_k)

    def _search_brute_force(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[str, float, EmbeddingItem]]:
        """暴力搜索"""
        results = []
        for key, item in self.store.items():
            if item.vector is not None:
                score = float(np.dot(query_vector, item.vector))
                results.append((key, score, item))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_all_keys(self) -> List[str]:
        """获取所有键"""
        return list(self.store.keys())

    def get_all_items(self) -> List[EmbeddingItem]:
        """获取所有项"""
        return list(self.store.values())

    def count(self) -> int:
        """获取存储项数量"""
        return len(self.store)

    def save_to_file(self) -> bool:
        """保存到文件"""
        with self._lock:
            try:
                store_data = {}
                for key, item in self.store.items():
                    store_data[key] = {
                        "key": item.key,
                        "vector": (
                            item.vector.tolist()
                            if item.vector is not None
                            else None
                        ),
                        "str": item.str,
                        "metadata": item.metadata,
                        "create_time": item.create_time,
                        "update_time": item.update_time,
                    }
                with open(self.store_file, "w", encoding="utf-8") as f:
                    json.dump(store_data, f, ensure_ascii=False, indent=2)
                if self.faiss_index is not None and FAISS_AVAILABLE:
                    faiss.write_index(self.faiss_index, self.index_file_path)
                with open(self.idx2hash_file_path, "wb") as f:
                    pickle.dump(self.idx2hash, f)
                self.dirty = False
                logger.info(
                    f"[{self.store_name}] 保存完成，共 {len(self.store)} 项"
                )
                return True
            except Exception as e:
                logger.error(f"[{self.store_name}] 保存失败: {e}")
                return False

    def load_from_file(self) -> bool:
        """从文件加载"""
        with self._lock:
            try:
                if os.path.exists(self.store_file):
                    with open(self.store_file, "r", encoding="utf-8") as f:
                        store_data = json.load(f)
                    self.store = {}
                    for key, data in store_data.items():
                        vector = None
                        if data.get("vector") is not None:
                            vector = np.array(data["vector"], dtype=np.float32)
                        item = EmbeddingItem(
                            key=data["key"],
                            vector=vector,
                            str=data.get("str", ""),
                            metadata=data.get("metadata", {}),
                            create_time=data.get("create_time", 0.0),
                            update_time=data.get("update_time", 0.0),
                        )
                        self.store[key] = item
                    logger.info(
                        f"[{self.store_name}] 从文件加载 {len(self.store)} 项"
                    )
                if FAISS_AVAILABLE and os.path.exists(self.index_file_path):
                    self.faiss_index = faiss.read_index(self.index_file_path)
                    if os.path.exists(self.idx2hash_file_path):
                        with open(self.idx2hash_file_path, "rb") as f:
                            self.idx2hash = pickle.load(
                                f
                            )  # 安全：仅加载自身写入的内部索引数据
                        self.hash2idx = {
                            v: k for k, v in self.idx2hash.items()
                        }
                else:
                    self._build_faiss_index()
                self.dirty = False
                return True
            except Exception as e:
                logger.error(f"[{self.store_name}] 加载失败: {e}")
                return False

    def clear(self):
        """清空存储"""
        with self._lock:
            self.store.clear()
            self.faiss_index = None
            self.idx2hash.clear()
            self.hash2idx.clear()
            self.dirty = True


class EmbeddingGenerator:
    """向量生成器，支持多线程批量生成"""

    def __init__(
        self,
        embedding_func: callable,
        max_workers: int = 4,
        chunk_size: int = 10,
    ):
        self.embedding_func = embedding_func
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self._executor: Optional[ThreadPoolExecutor] = None

    def _get_executor(self) -> ThreadPoolExecutor:
        """获取线程池执行器"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def generate_single(self, text: str) -> Optional[np.ndarray]:
        """生成单个文本的向量"""
        try:
            return self.embedding_func(text)
        except Exception as e:
            logger.error(f"生成向量失败: {e}")
            return None

    def generate_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """批量生成向量"""
        results = [None] * len(texts)
        total = len(texts)
        if total == 0:
            return results
        executor = self._get_executor()

        def process_chunk(chunk_indices: List[int]) -> Dict[int, np.ndarray]:
            chunk_results = {}
            for idx in chunk_indices:
                try:
                    vector = self.embedding_func(texts[idx])
                    chunk_results[idx] = vector
                except Exception as e:
                    logger.error(f"生成第 {idx} 个向量失败: {e}")
                    chunk_results[idx] = None
            return chunk_results

        chunks = []
        for i in range(0, total, self.chunk_size):
            chunk_indices = list(range(i, min(i + self.chunk_size, total)))
            chunks.append(chunk_indices)
        futures = []
        for chunk in chunks:
            future = executor.submit(process_chunk, chunk)
            futures.append(future)
        completed = 0
        for future in futures:
            chunk_results = future.result()
            for idx, vector in chunk_results.items():
                results[idx] = vector
            completed += len(chunk_results)
            if show_progress:
                logger.info(f"[EmbeddingGenerator] 进度: {completed}/{total}")
        return results

    async def generate_batch_async(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """异步批量生成向量"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate_batch(texts, show_progress)
        )

    def shutdown(self):
        """关闭线程池"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


class EmbeddingManager:
    """向量管理器，管理段落、实体、关系三种向量存储"""

    def __init__(
        self,
        data_dir: str = "",
        dimension: int = 1024,
        max_workers: int = 4,
        chunk_size: int = 10,
    ):
        self.data_dir = data_dir or self._get_default_data_dir()
        self.dimension = dimension
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        os.makedirs(self.data_dir, exist_ok=True)
        self.paragraphs_embedding_store = EmbeddingStore(
            store_name="paragraphs",
            data_dir=self.data_dir,
            dimension=dimension,
        )
        self.entities_embedding_store = EmbeddingStore(
            store_name="entities",
            data_dir=self.data_dir,
            dimension=dimension,
        )
        self.relation_embedding_store = EmbeddingStore(
            store_name="relations",
            data_dir=self.data_dir,
            dimension=dimension,
        )
        self.stored_pg_hashes: set = set()
        self._embedding_generator: Optional[EmbeddingGenerator] = None
        self._embedding_func: Optional[callable] = None

    def _get_default_data_dir(self) -> str:
        """获取默认数据目录"""
        current = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(current, "..", "..", ".."))
        return os.path.join(root, "data", "knowledge", "embeddings")

    def set_embedding_func(self, func: callable):
        """设置向量生成函数"""
        self._embedding_func = func
        self._embedding_generator = EmbeddingGenerator(
            embedding_func=func,
            max_workers=self.max_workers,
            chunk_size=self.chunk_size,
        )

    def _get_embedding_generator(self) -> EmbeddingGenerator:
        """获取向量生成器"""
        if self._embedding_generator is None:
            raise RuntimeError(
                "未设置向量生成函数，请先调用 set_embedding_func()"
            )
        return self._embedding_generator

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """生成单个文本的向量"""
        generator = self._get_embedding_generator()
        return generator.generate_single(text)

    def generate_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """批量生成向量"""
        generator = self._get_embedding_generator()
        return generator.generate_batch(texts, show_progress)

    async def generate_embeddings_batch_async(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[Optional[np.ndarray]]:
        """异步批量生成向量"""
        generator = self._get_embedding_generator()
        return await generator.generate_batch_async(texts, show_progress)

    def store_paragraph(
        self,
        pg_hash: str,
        text: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """存储段落向量"""
        if vector is None:
            vector = self.generate_embedding(text)
            if vector is None:
                return False
        key = f"paragraph-{pg_hash}"
        success = self.paragraphs_embedding_store.add_item(
            key=key,
            vector=vector,
            text=text,
            metadata=metadata or {},
        )
        if success:
            self.stored_pg_hashes.add(key)
        return success

    def store_entity(
        self,
        entity_name: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """存储实体向量"""
        if vector is None:
            vector = self.generate_embedding(entity_name)
            if vector is None:
                return False
        key = f"entity-{entity_name}"
        return self.entities_embedding_store.add_item(
            key=key,
            vector=vector,
            text=entity_name,
            metadata=metadata or {},
        )

    def store_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """存储关系向量"""
        relation_text = f"{subject} {predicate} {obj}"
        if vector is None:
            vector = self.generate_embedding(relation_text)
            if vector is None:
                return False
        key = f"relation-{subject}-{predicate}-{obj}"
        return self.relation_embedding_store.add_item(
            key=key,
            vector=vector,
            text=relation_text,
            metadata=metadata
            or {"subject": subject, "predicate": predicate, "object": obj},
        )

    def store_new_data_set(
        self,
        raw_paragraphs: Dict[str, str],
        triple_list_data: Dict[str, List[List[str]]],
        show_progress: bool = True,
    ) -> Dict[str, int]:
        """存储新数据集"""
        stats = {"paragraphs": 0, "entities": 0, "relations": 0}
        pg_hashes = list(raw_paragraphs.keys())
        pg_texts = [raw_paragraphs[h] for h in pg_hashes]
        logger.info(
            f"[EmbeddingManager] 正在生成 {len(pg_texts)} 个段落向量..."
        )
        pg_vectors = self.generate_embeddings_batch(pg_texts, show_progress)
        for i, pg_hash in enumerate(pg_hashes):
            if pg_vectors[i] is not None:
                if self.store_paragraph(pg_hash, pg_texts[i], pg_vectors[i]):
                    stats["paragraphs"] += 1
        all_entities = set()
        all_relations = []
        for pg_hash, triples in triple_list_data.items():
            for triple in triples:
                if len(triple) == 3:
                    subject, predicate, obj = triple
                    all_entities.add(subject)
                    all_entities.add(obj)
                    all_relations.append((subject, predicate, obj))
        entity_list = list(all_entities)
        logger.info(
            f"[EmbeddingManager] 正在生成 {len(entity_list)} 个实体向量..."
        )
        entity_vectors = self.generate_embeddings_batch(
            entity_list, show_progress
        )
        for i, entity in enumerate(entity_list):
            if entity_vectors[i] is not None:
                if self.store_entity(entity, entity_vectors[i]):
                    stats["entities"] += 1
        logger.info(
            f"[EmbeddingManager] 正在生成 {len(all_relations)} 个关系向量..."
        )
        relation_texts = [f"{s} {p} {o}" for s, p, o in all_relations]
        relation_vectors = self.generate_embeddings_batch(
            relation_texts, show_progress
        )
        for i, (subject, predicate, obj) in enumerate(all_relations):
            if relation_vectors[i] is not None:
                if self.store_relation(
                    subject, predicate, obj, relation_vectors[i]
                ):
                    stats["relations"] += 1
        logger.info(f"[EmbeddingManager] 数据存储完成: {stats}")
        return stats

    def search_paragraphs(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float, EmbeddingItem]]:
        """搜索段落"""
        return self.paragraphs_embedding_store.search_top_k(
            query_vector, top_k
        )

    def search_entities(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float, EmbeddingItem]]:
        """搜索实体"""
        return self.entities_embedding_store.search_top_k(query_vector, top_k)

    def search_relations(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float, EmbeddingItem]]:
        """搜索关系"""
        return self.relation_embedding_store.search_top_k(query_vector, top_k)

    def rebuild_faiss_index(self):
        """重建所有 FAISS 索引"""
        self.paragraphs_embedding_store.rebuild_faiss_index()
        self.entities_embedding_store.rebuild_faiss_index()
        self.relation_embedding_store.rebuild_faiss_index()
        logger.info("[EmbeddingManager] FAISS 索引重建完成")

    def save_to_file(self) -> bool:
        """保存所有数据到文件"""
        success = True
        success &= self.paragraphs_embedding_store.save_to_file()
        success &= self.entities_embedding_store.save_to_file()
        success &= self.relation_embedding_store.save_to_file()
        hashes_file = os.path.join(self.data_dir, "stored_pg_hashes.json")
        try:
            with open(hashes_file, "w", encoding="utf-8") as f:
                json.dump(list(self.stored_pg_hashes), f)
        except Exception as e:
            logger.error(f"保存段落哈希集合失败: {e}")
            success = False
        return success

    def load_from_file(self) -> bool:
        """从文件加载所有数据"""
        success = True
        success &= self.paragraphs_embedding_store.load_from_file()
        success &= self.entities_embedding_store.load_from_file()
        success &= self.relation_embedding_store.load_from_file()
        self.stored_pg_hashes = set(
            self.paragraphs_embedding_store.get_all_keys()
        )
        hashes_file = os.path.join(self.data_dir, "stored_pg_hashes.json")
        if os.path.exists(hashes_file):
            try:
                with open(hashes_file, "r", encoding="utf-8") as f:
                    self.stored_pg_hashes = set(json.load(f))
            except Exception as e:
                logger.warning(f"加载段落哈希集合失败: {e}")
        logger.info(
            f"[EmbeddingManager] 加载完成: {len(self.stored_pg_hashes)} 个段落哈希"
        )
        return success

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "paragraphs": self.paragraphs_embedding_store.count(),
            "entities": self.entities_embedding_store.count(),
            "relations": self.relation_embedding_store.count(),
            "stored_pg_hashes": len(self.stored_pg_hashes),
        }

    def clear_all(self):
        """清空所有数据"""
        self.paragraphs_embedding_store.clear()
        self.entities_embedding_store.clear()
        self.relation_embedding_store.clear()
        self.stored_pg_hashes.clear()
        logger.info("[EmbeddingManager] 所有数据已清空")

    def shutdown(self):
        """关闭资源"""
        if self._embedding_generator is not None:
            self._embedding_generator.shutdown()


def check_embedding_model_consistency(
    embedding_func: callable,
    test_text: str = "测试文本",
    expected_dimension: Optional[int] = None,
) -> Tuple[bool, int]:
    """检查向量模型一致性"""
    try:
        vector = embedding_func(test_text)
        if vector is None:
            return False, 0
        actual_dimension = vector.shape[0]
        if (
            expected_dimension is not None
            and actual_dimension != expected_dimension
        ):
            logger.warning(
                f"向量维度不一致: 期望 {expected_dimension}, 实际 {actual_dimension}"
            )
            return False, actual_dimension
        return True, actual_dimension
    except Exception as e:
        logger.error(f"检查向量模型失败: {e}")
        return False, 0

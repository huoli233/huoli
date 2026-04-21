import os
from typing import Optional
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("knowledge")

_qa_manager = None
_embed_manager = None
_kg_manager = None


def get_qa_manager():
    return _qa_manager


def get_embed_manager():
    return _embed_manager


def get_kg_manager():
    return _kg_manager


def lpmm_start_up():
    global _qa_manager, _embed_manager, _kg_manager
    if not global_config.lpmm_knowledge.enable:
        logger.info("LPMM知识库未启用，跳过初始化")
        return
    try:
        from src.chat.knowledge.embedding_store import EmbeddingManager
        from src.chat.knowledge.kg_manager import KGManager
        from src.chat.knowledge.qa_manager import QAManager
        _embed_manager = EmbeddingManager()
        _embed_manager.load_from_file()
        _embed_manager.rebuild_faiss_index()
        logger.info("EmbeddingManager 初始化完成")
        _kg_manager = KGManager()
        try:
            _kg_manager.load_from_file()
            logger.info("KGManager 初始化完成")
        except FileNotFoundError:
            logger.warning("KG数据文件不存在，将使用空知识图谱")
        _qa_manager = QAManager(_embed_manager, _kg_manager)
        logger.info("LPMM知识库初始化完成")
    except Exception as e:
        logger.error(f"LPMM知识库初始化失败: {e}")
        _qa_manager = None

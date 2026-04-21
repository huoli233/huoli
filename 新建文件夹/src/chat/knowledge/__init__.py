import asyncio
import threading
from typing import Optional

import numpy as np

from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("knowledge")

_qa_manager = None
_embed_manager = None
_kg_manager = None
_embedding_loop = None
_embedding_thread = None
_embedding_loop_lock = threading.Lock()


def get_qa_manager():
    return _qa_manager


def get_embed_manager():
    return _embed_manager


def get_kg_manager():
    return _kg_manager


def _run_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_embedding_loop() -> asyncio.AbstractEventLoop:
    global _embedding_loop, _embedding_thread

    with _embedding_loop_lock:
        if _embedding_loop is not None and _embedding_loop.is_running():
            return _embedding_loop

        _embedding_loop = asyncio.new_event_loop()
        _embedding_thread = threading.Thread(
            target=_run_background_loop,
            args=(_embedding_loop,),
            name="lpmm-embedding-loop",
            daemon=True,
        )
        _embedding_thread.start()
        return _embedding_loop


async def _request_embedding_async(text: str) -> Optional[np.ndarray]:
    from src.chat.utils.utils import get_embedding

    vector = await get_embedding(text, request_type="lpmm_embedding")
    if not vector:
        return None
    return np.asarray(vector, dtype=np.float32)


def _request_embedding_sync(text: str) -> Optional[np.ndarray]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_request_embedding_async(text))

    future = asyncio.run_coroutine_threadsafe(
        _request_embedding_async(text), _ensure_embedding_loop()
    )
    return future.result()


def lpmm_start_up():
    global _qa_manager, _embed_manager, _kg_manager

    if not global_config.memory.enable_lpmm:
        logger.info("LPMM knowledge disabled, skip startup")
        return

    if (
        _qa_manager is not None
        and _embed_manager is not None
        and _kg_manager is not None
    ):
        return

    try:
        from src.chat.knowledge.embedding_store import EmbeddingManager
        from src.chat.knowledge.kg_manager import KGManager
        from src.chat.knowledge.lpmm_ops import initialize_lpmm
        from src.chat.knowledge.qa_manager import QAManager

        lpmm_config = global_config.lpmm_knowledge
        embed_manager = EmbeddingManager(
            dimension=lpmm_config.embedding_dimension,
            max_workers=lpmm_config.max_embedding_workers,
            chunk_size=lpmm_config.embedding_chunk_size,
        )
        embed_manager.set_embedding_func(_request_embedding_sync)
        embed_manager.load_from_file()
        embed_manager.rebuild_faiss_index()

        kg_manager = KGManager()
        kg_manager.load_from_file()

        qa_manager = QAManager(embed_manager, kg_manager)
        initialize_lpmm(embed_manager, kg_manager, qa_manager)

        _embed_manager = embed_manager
        _kg_manager = kg_manager
        _qa_manager = qa_manager
        logger.info("LPMM knowledge startup complete")
    except Exception as exc:
        logger.error(f"LPMM knowledge startup failed: {exc}", exc_info=True)
        _qa_manager = None
        _embed_manager = None
        _kg_manager = None


__all__ = [
    "get_embed_manager",
    "get_kg_manager",
    "get_qa_manager",
    "lpmm_start_up",
]

raise DeprecationWarning(
    "MemoryActiveManager is not used yet, please do not import it"
)
try:
    from .lpmmconfig import (
        global_config,
    )  # noqa: 模块依赖：lpmmconfig（LPMM配置）
except ImportError:
    global_config = None  # noqa: 模块不可用时使用None占位
try:
    from .embedding_store import (
        EmbeddingManager,
    )  # noqa: 模块依赖：embedding_store（嵌入存储管理器）
except ImportError:
    EmbeddingManager = None  # noqa: 模块不可用时使用None占位
try:
    from .llm_client import (
        LLMClient,
    )  # noqa: 模块依赖：llm_client（LLM客户端）
except ImportError:
    LLMClient = None  # noqa: 模块不可用时使用None占位
try:
    from .utils.dyn_topk import (
        dyn_select_top_k,
    )  # noqa: 模块依赖：utils.dyn_topk（动态TopK选择）
except ImportError:
    dyn_select_top_k = None  # noqa: 模块不可用时使用None占位


class MemoryActiveManager:
    def __init__(
        self,
        embed_manager: EmbeddingManager,
        llm_client_embedding: LLMClient,
    ):
        self.embed_manager = embed_manager
        self.embedding_client = llm_client_embedding

    def get_activation(self, question: str) -> float:
        """获取记忆激活度"""
        # 生成问题的Embedding
        question_embedding = self.embedding_client.send_embedding_request(
            "text-embedding", question
        )
        # 查询关系库中的相似度
        rel_search_res = (
            self.embed_manager.relation_embedding_store.search_top_k(
                question_embedding, 10
            )
        )

        # 动态过滤阈值
        rel_scores = dyn_select_top_k(rel_search_res, 0.5, 1.0)
        if (
            rel_scores[0][1]
            < global_config["qa"]["params"]["relation_threshold"]
        ):
            # 未找到相关关系
            return 0.0

        # 计算激活度
        activation = sum([item[2] for item in rel_scores]) * 10

        return activation

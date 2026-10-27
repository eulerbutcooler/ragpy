
import logging
import threading

from llama_index.core.postprocessor.types import BaseNodePostprocessor

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Global lock to serialize GPU cross-encoder inference across concurrent asyncio threads
# Prevents PyTorch "RuntimeError: expected scalar type Float but found Half" caused by
# concurrent calls to self.model.half() in FlagEmbedding.
_RERANK_LOCK = threading.Lock()


def safe_postprocess_nodes(
    reranker: BaseNodePostprocessor,
    nodes: list,
    query_bundle=None,
    top_n: int = 6,
) -> list:
    """Thread-safe reranker runner preventing concurrent PyTorch model state mutations."""
    if not nodes:
        return []
    if len(nodes) <= top_n:
        return nodes
    with _RERANK_LOCK:
        try:
            return reranker.postprocess_nodes(nodes, query_bundle)
        except Exception as err:
            logger.warning("reranker_call_failed fallback_to_unranked err=%s", err)
            return nodes[:top_n]


def get_reranker(top_n: int | None = None) -> BaseNodePostprocessor:
    if top_n is None:
        top_n = settings.reranker_top_n
    try:
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        return FlagEmbeddingReranker(
            model=settings.reranker_model_name,
            top_n=top_n,
            use_fp16=True,
        )
    except Exception as e:
        logger.error("reranker_load_failed error=%s", e)
        raise

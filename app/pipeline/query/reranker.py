from __future__ import annotations

import logging
import threading

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from pydantic import ConfigDict

from app.config.settings import settings
from app.llm.factory import normalize_openai_base_url

logger = logging.getLogger(__name__)

# Serialize local cross-encoder inference. The torch model is shared across
# threads, so keep calls serialized.
_RERANK_LOCK = threading.Lock()
_BASE_RERANKER_LOCK = threading.Lock()
_BASE_RERANKER: BaseNodePostprocessor | None = None


class _CrossEncoderReranker(BaseNodePostprocessor):
    """Local reranker backed by sentence-transformers' CrossEncoder.

    This is used instead of FlagEmbedding because FlagEmbedding still imports
    transformer internals that were removed in Transformers 5.x.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: object
    top_n: int

    def __init__(self, model_name: str, top_n: int) -> None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            model_name,
            max_length=settings.reranker_max_length,
            device=settings.embed_device,
        )
        super().__init__(model=model, top_n=top_n)

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return []
        from llama_index.core.schema import NodeWithScore

        query = query_bundle.query_str if query_bundle else ""
        pairs = [(query, node.node.get_content()) for node in nodes]
        scores = self.model.predict(  # type: ignore[attr-defined]
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(nodes, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            NodeWithScore(node=node.node, score=float(score))
            for node, score in ranked[: self.top_n]
        ]


def _build_base_reranker() -> BaseNodePostprocessor:
    """Build the one shared reranker at the largest top_n any caller needs."""
    max_top_n = settings.reranker_max_top_n

    if settings.reranker_backend == "remote":
        from app.llm.remote import RemoteReranker

        base_url = settings.reranker_base_url or settings.llm_base_url
        return RemoteReranker(
            base_url=normalize_openai_base_url(base_url),
            api_key=settings.llm_api_key,
            model=settings.reranker_model_name,
            top_n=max_top_n,
            timeout=settings.reranker_timeout,
        )

    return _CrossEncoderReranker(
        model_name=settings.reranker_model_name,
        top_n=max_top_n,
    )


def get_reranker(top_n: int | None = None) -> BaseNodePostprocessor:
    """Return the single shared underlying reranker.

    ``top_n`` is accepted for API compatibility. The underlying model is always
    built at ``settings.reranker_max_top_n``; callers slice the ranked list to
    the size they need. This avoids loading multiple copies of the model.
    """
    global _BASE_RERANKER
    if _BASE_RERANKER is not None:
        return _BASE_RERANKER

    with _BASE_RERANKER_LOCK:
        if _BASE_RERANKER is None:
            try:
                _BASE_RERANKER = _build_base_reranker()
            except Exception as err:
                logger.error("reranker_load_failed error=%s", err)
                raise
    return _BASE_RERANKER


def safe_postprocess_nodes(
    reranker: BaseNodePostprocessor,
    nodes: list,
    query_bundle=None,
    top_n: int = 6,
) -> list:
    """Lock, rank, slice, and fall back gracefully on reranker failure."""
    if not nodes or top_n <= 0:
        return []
    if len(nodes) <= top_n:
        return nodes[:top_n]
    if top_n > settings.reranker_max_top_n:
        logger.warning(
            "reranker_top_n_exceeds_max requested=%d max=%d",
            top_n,
            settings.reranker_max_top_n,
        )

    try:
        if settings.reranker_backend == "remote":
            # The Modal gateway serializes GPU rerank calls itself.
            ranked = reranker.postprocess_nodes(nodes, query_bundle)
        else:
            with _RERANK_LOCK:
                ranked = reranker.postprocess_nodes(nodes, query_bundle)
        return ranked[:top_n]
    except Exception as err:
        logger.warning("reranker_call_failed fallback_to_unranked err=%s", err)
        return nodes[:top_n]


class SafeReranker(BaseNodePostprocessor):
    """Postprocessor wrapper that gives chat the same safety as quiz."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    underlying: BaseNodePostprocessor
    top_n: int = 3

    def _postprocess_nodes(self, nodes, query_bundle=None):
        return safe_postprocess_nodes(
            self.underlying,
            nodes,
            query_bundle,
            self.top_n,
        )

"""Hybrid retriever: combines dense vector + BM25 using Reciprocal Rank Fusion.

RRF formula: score(d) = Σ 1 / (k + rank(r, d))
where k = 60 (industry standard), r iterates over each ranking list.
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.config.settings import settings
from app.pipeline.query.dense_retriever import get_retriever
from app.pipeline.query.sparse_retriever import bm25_retrieve

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]],
    k: int = 60,
) -> list[NodeWithScore]:
    """Merge multiple ranked lists using RRF."""
    scores: dict[str, float] = defaultdict(float)
    node_map: dict[str, NodeWithScore] = {}

    for ranked_list in ranked_lists:
        for rank, node_with_score in enumerate(ranked_list, start=1):
            node_id = node_with_score.node.node_id
            scores[node_id] += 1.0 / (k + rank)
            # Keep the node reference (prefer the one with highest original score)
            if node_id not in node_map or (node_with_score.score or 0) > (
                node_map[node_id].score or 0
            ):
                node_map[node_id] = node_with_score

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda nid: scores[nid], reverse=True)

    return [
        NodeWithScore(node=node_map[nid].node, score=scores[nid])
        for nid in sorted_ids
    ]


class HybridRetriever(BaseRetriever):
    """Retriever that fuses dense vector search + BM25 via RRF."""

    def __init__(self, course_ids: list[str], top_k: int | None = None):
        super().__init__()
        self._course_ids = course_ids
        self._top_k = top_k or settings.retrieval_top_k
        self._rrf_k = settings.rrf_k
        self._dense_retriever = get_retriever(course_ids=course_ids, top_k=self._top_k)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_str = query_bundle.query_str

        # Run dense + BM25 in parallel (they're independent)
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(self._dense_retriever.retrieve, query_str)
            bm25_future = pool.submit(
                bm25_retrieve,
                query=query_str,
                course_ids=self._course_ids,
                top_k=self._top_k,
            )
            dense_results = dense_future.result()
            bm25_results = bm25_future.result()

        logger.info(
            "hybrid_retrieve courses=%s dense=%d bm25=%d",
            self._course_ids,
            len(dense_results),
            len(bm25_results),
        )

        # Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(
            [dense_results, bm25_results], k=self._rrf_k
        )

        # Return top_k fused results (the reranker will further filter)
        return fused[: self._top_k]


from llama_index.core.query_engine import RetrieverQueryEngine

from app.config.settings import settings
from app.pipeline.query.reranker import SafeReranker, get_reranker
from app.pipeline.query.synthesizer import get_synthesizer


def get_query_engine(course_ids: list[str], streaming: bool = False) -> RetrieverQueryEngine:
    if settings.hybrid_search_enabled:
        from app.pipeline.query.full_retriever import HybridRetriever

        retriever = HybridRetriever(course_ids=course_ids)
    else:
        from app.pipeline.query.dense_retriever import get_retriever

        retriever = get_retriever(course_ids=course_ids)

    # Shared model + per-request slice; SafeReranker adds locking and fallback.
    reranker = SafeReranker(
        underlying=get_reranker(),
        top_n=settings.reranker_top_n,
    )
    synthesizer = get_synthesizer(streaming=streaming)

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_synthesizer=synthesizer,
    )

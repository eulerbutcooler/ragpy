"""Retrieve figure captions relevant to a topic."""

from __future__ import annotations

import asyncio
import logging

from qdrant_client.http import models as qdrant_models

from app.config.settings import settings
from app.llm.factory import get_embed_model
from app.store.qdrant import get_sync_client

logger = logging.getLogger(__name__)


async def retrieve_figures(course_id: str, query: str, limit: int = 8) -> list[dict]:
    if not course_id or not query.strip():
        return []

    embedder = get_embed_model()
    try:
        vector = await asyncio.to_thread(embedder.get_text_embedding, query)
    except Exception:  # noqa: BLE001 - retrieval is optional
        logger.exception("figure_query_embed_failed course_id=%s", course_id)
        return []

    client = get_sync_client()
    flt = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="course_id", match=qdrant_models.MatchValue(value=course_id)
            ),
            qdrant_models.FieldCondition(
                key="content_type", match=qdrant_models.MatchValue(value="figure")
            ),
        ]
    )

    def _search():
        return client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=vector,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )

    try:
        result = await asyncio.to_thread(_search)
    except Exception:  # noqa: BLE001 - older client fallback
        def _legacy():
            return client.search(
                collection_name=settings.qdrant_collection_name,
                query_vector=vector,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )

        try:
            points = await asyncio.to_thread(_legacy)
        except Exception:
            logger.exception("figure_search_failed course_id=%s", course_id)
            return []
        result = type("Result", (), {"points": points})()

    figures: list[dict] = []
    for point in result.points:
        payload = point.payload or {}
        figure_id = payload.get("figure_id")
        caption = payload.get("caption")
        if not figure_id or not caption:
            continue
        figures.append(
            {
                "id": str(figure_id),
                "caption": str(caption),
                "page_no": payload.get("page_no"),
                "file_name": payload.get("file_name"),
            }
        )
    logger.info("figures_retrieved course_id=%s query=%r count=%d", course_id, query[:60], len(figures))
    return figures
